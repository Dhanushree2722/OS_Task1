import java.util.LinkedList;
import java.util.Queue;
import java.util.Scanner;

// Main class for the warehouse simulation.
// Asks for shelf capacity, number of packers, number of couriers and parcels per packer,
// then starts the packer and courier threads and waits for all of them to finish.
// Packers and couriers never talk to each other directly; everything goes through Shelf.
public class WarehouseSimulation {

    public static void main(String[] args) throws InterruptedException {
        Scanner sc = new Scanner(System.in);

        // simulation parameters
        System.out.print(" Enter Shelf capacity: ");
        int capacity = sc.nextInt();

        System.out.print("Enter Number of packers: ");
        int packers = sc.nextInt();

        System.out.print("Enter Number of couriers: ");
        int couriers = sc.nextInt();

        System.out.print("Enter Parcels per packer: ");
        int parcelsPerPacker = sc.nextInt();
        sc.close();

        Shelf shelf = new Shelf(capacity);

        Thread[] producers = new Thread[packers];
        Thread[] consumers = new Thread[couriers];

        // packers (producers)
        for (int i = 0; i < packers; i++) {
            producers[i] = new Thread(
                new Packer(i + 1, shelf, parcelsPerPacker)
            );
            producers[i].start();
        }

        // couriers (consumers)
        for (int i = 0; i < couriers; i++) {
            consumers[i] = new Thread(
                new Courier(i + 1, shelf)
            );
            consumers[i].start();
        }

        // wait for the packers first
        for (Thread t : producers) {
            t.join();
        }

        // now nothing else will be produced, so tell the shelf
        shelf.close();

        // and wait until the couriers have delivered whatever is left
        for (Thread t : consumers) {
            t.join();
        }

        System.out.println("Simulation completed.");
    }
}

/* Shelf is the shared resource. Packers put parcels on it, couriers take them off.
   It is a queue with a capacity limit, and put/take are synchronized so only one thread
   touches the queue at a time. close() marks that no more parcels will be added. */

class Shelf {

    private final Queue<String> queue = new LinkedList<>();
    private final int capacity;
    private boolean open = true;

    Shelf(int capacity) {
        this.capacity = capacity;
    }

    // producer side
    synchronized void put(String parcel) throws InterruptedException {

        while (queue.size() == capacity) {
            wait();
        }

        queue.add(parcel);
        System.out.println("Produced: " + parcel);

        notifyAll();
    }

    // consumer side; returns null when the shelf is closed and empty
    synchronized String take() throws InterruptedException {

        while (queue.isEmpty() && open) {
            wait();
        }

        if (queue.isEmpty()) {
            return null;
        }

        String parcel = queue.remove();

        notifyAll();

        return parcel;
    }

    // no more parcels coming; wake up any waiting couriers
    synchronized void close() {
        open = false;
        notifyAll();
    }
}

// Packer is the producer. Each one has an id, the shared shelf, and how many parcels
// to make. It packs one parcel at a time and sleeps a bit in between.
class Packer implements Runnable {

    private final int id;
    private final Shelf shelf;
    private final int parcelsToProduce;

    Packer(int id, Shelf shelf, int parcelsToProduce) {
        this.id = id;
        this.shelf = shelf;
        this.parcelsToProduce = parcelsToProduce;
    }

    @Override
    public void run() {
        try {
            for (int i = 0; i < parcelsToProduce; i++) {
                String parcel = "Parcel " + (i + 1) + " from Packer " + id;
                // blocks if the shelf is full
                shelf.put(parcel);
                Thread.sleep(100); // time taken to pack a parcel
            }
        } catch (InterruptedException e) {
            // keep the interrupted flag set and just stop
            Thread.currentThread().interrupt();
        }
    }
}

// Courier is the consumer. It keeps taking parcels off the shelf and "delivering" them
// (printing a line and sleeping) until take() returns null.
class Courier implements Runnable {

    private final int id;
    private final Shelf shelf;

    Courier(int id, Shelf shelf) {
        this.id = id;
        this.shelf = shelf;
    }

    @Override
    public void run() {
        try {
            while (true) {
                // blocks if the shelf is empty but still open
                String parcel = shelf.take();
                if (parcel == null) {
                    break; // shelf closed and nothing left
                }
                System.out.println("Delivered: " + parcel + " by Courier " + id);
                Thread.sleep(150); // time taken to deliver a parcel
            }
        } catch (InterruptedException e) {
            // keep the interrupted flag set and just stop
            Thread.currentThread().interrupt();
        }
    }
}